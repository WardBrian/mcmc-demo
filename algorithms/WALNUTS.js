"use strict";

// based on Bob Carpenter's C++ implementation.

MCMC.registerAlgorithm("WALNUTS", {
  description: "Sub-orbit adapation for the No-U-Turn Sampler",

  about: function () {
    // TODO: add real link
    window.open("https://arxiv.org/abs/2506.18746");
  },

  init: function (self) {
    self.dt = 0.4;
    self.maxError = 0.8;
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.add(self, "dt", 0.025, 1.2).step(0.025).name("Leapfrog &Delta;t");
    folder.add(self, "maxError", 0.01, 4).step(0.01).name("Max error");
    folder.open();
  },

  step: function (self, visualizer) {
    var trajectory = [];

    function Span(q_bk, p_bk, grad_bk, logp_bk, q_fw, p_fw, grad_fw, logp_fw, q_sel, logp_sel) {
      this.theta_bk_ = q_bk;
      this.rho_bk_ = p_bk;
      this.grad_theta_bk_ = grad_bk;
      this.logp_bk_ = logp_bk;
      this.theta_fw_ = q_fw;
      this.rho_fw_ = p_fw;
      this.grad_theta_fw_ = grad_fw;
      this.logp_fw_ = logp_fw;
      this.theta_select_ = q_sel;
      this.logp_ = logp_sel;
    }

    function make_leaf_span(theta, rho, grad_theta, logp) {
      return new Span(
        theta.copy(), rho.copy(), grad_theta.copy(), logp,
        theta.copy(), rho.copy(), grad_theta.copy(), logp,
        theta.copy(), logp
      );
    }

    function make_combined_span(span_bk, span_fw, theta_select, logp_total) {
      return new Span(
        span_bk.theta_bk_.copy(), span_bk.rho_bk_.copy(), span_bk.grad_theta_bk_.copy(), span_bk.logp_bk_,
        span_fw.theta_fw_.copy(), span_fw.rho_fw_.copy(), span_fw.grad_theta_fw_.copy(), span_fw.logp_fw_,
        theta_select.copy(), logp_total
      );
    }

    function log_sum_exp(x, y) {
      var m = Math.max(x, y);
      return m + Math.log(Math.exp(x - m) + Math.exp(y - m));
    }

    function within_tolerance(step, numSteps, q, p, grad, logp) {
      let q_next = q.copy();
      let p_next = p.copy();
      let grad_next = grad.copy();
      let halfStep = 0.5 * step;
      let logp_initial = logp;
      for (let n = 0; n < numSteps; ++n) {
        p_next.increment(grad_next.scale(halfStep));
        q_next.increment(p_next.scale(step));
        grad_next = self.gradLogDensity(q_next);
        p_next.increment(grad_next.scale(halfStep));
      }
      let logp_next = self.logDensity(q_next) - p_next.norm2() / 2;
      return Math.abs(logp_next - logp_initial) <= self.maxError;
    }

    function reversible(step, numSteps, q, p, grad, logp) {
      if (numSteps == 1) return true;
      let q_next, p_next, grad_next;
      while (numSteps >= 2) {
        q_next = q.copy();
        p_next = p.copy().scale(-1); // negate momentum
        grad_next = grad.copy();
        numSteps = Math.floor(numSteps / 2);
        step = step * 2;
        // If the backwards trajectory is within tolerance, it's not reversible
        if (within_tolerance(step, numSteps, q_next, p_next, grad_next, logp)) {
          return false;
        }
      }
      return true;
    }

    function macro_step(theta, rho, grad_theta, logp_theta, v, trajectory) {
      var theta0 = theta.copy();
      var rho0 = rho.copy();
      var grad_theta0 = grad_theta.copy();
      var logp_theta0 = logp_theta;
      var maxHalvings = 10;
      var macroStepReturn = true;
      var theta_next, rho_next, grad_theta_next, logp_next;
      var finalLeapfrogs = [];
      var step = v * self.dt;

      for (var halvings = 0, num_steps = 1; halvings < maxHalvings; ++halvings, num_steps *= 2, step *= 0.5) {
        theta_next = theta0.copy();
        rho_next = rho0.copy();
        grad_theta_next = grad_theta0.copy();
        var leapfrogs = [];
        var half_step = step / 2;
        for (var n = 0; n < num_steps; ++n) {
          rho_next.increment(grad_theta_next.scale(half_step));
          theta_next.increment(rho_next.scale(step));
          grad_theta_next = self.gradLogDensity(theta_next);
          rho_next.increment(grad_theta_next.scale(half_step));
          leapfrogs.push({
            type: "leapfrog",
            halvings: halvings,
            from: (n === 0) ? theta0.copy() : leapfrogs[leapfrogs.length - 1].to.copy(),
            to: theta_next.copy(),
            stepSize: step,
            step: n
          });
        }
        logp_next = self.logDensity(theta_next) - rho_next.norm2() / 2;
        if (Math.abs(logp_theta0 - logp_next) <= self.maxError) {
          var rev = reversible(step, num_steps, theta_next, rho_next, grad_theta_next, logp_next);
          if (rev) {
            // Trajectory is reversible, this is success
            macroStepReturn = false; // false means success in C++
            finalLeapfrogs = leapfrogs;
            break;
          }
        }
        macroStepReturn = true;
      }
      if (macroStepReturn) {
        return null; // Failed to find valid trajectory
      }

      for (var lf of finalLeapfrogs) trajectory.push(lf);
      trajectory.push({
        type: "accept", // Success case
        from: theta0.copy(),
        to: theta_next.copy(),
      });
      return make_leaf_span(
        theta_next, rho_next, grad_theta_next, logp_next
      );
    }

    function uturn(span1, span2, direction) {
      var span_bk, span_fw;
      if (direction === 1) { // Forward
        span_bk = span1;
        span_fw = span2;
      } else { // Backward
        span_bk = span2;
        span_fw = span1;
      }
      var scaled_diff = span_fw.theta_fw_.subtract(span_bk.theta_bk_);
      return (span_fw.rho_fw_.dot(scaled_diff) < 0) || (span_bk.rho_bk_.dot(scaled_diff) < 0);
    }

    function combine(span1, span2, useBarker, direction) {
      var logp1 = span1.logp_;
      var logp2 = span2.logp_;
      var logp_total = log_sum_exp(logp1, logp2);
      var log_denominator;
      if (useBarker) {
        log_denominator = logp_total;
      } else { // Metropolis
        log_denominator = logp1;
      }
      var update_logprob = logp2 - log_denominator;
      var update = Math.log(Math.random()) < update_logprob;
      var theta_select = update ? span2.theta_select_ : span1.theta_select_;

      var span_bk, span_fw;
      if (direction === 1) { // Forward
        span_bk = span1;
        span_fw = span2;
      } else { // Backward
        span_bk = span2;
        span_fw = span1;
      }
      return make_combined_span(
        span_bk, span_fw, theta_select, logp_total
      );
    }

    function build_leaf(span, v, trajectory) {
      var theta = v === 1 ? span.theta_fw_ : span.theta_bk_;
      var rho = v === 1 ? span.rho_fw_ : span.rho_bk_;
      var grad_theta = v === 1 ? span.grad_theta_fw_ : span.grad_theta_bk_;
      var logp_theta = v === 1 ? span.logp_fw_ : span.logp_bk_;
      var result = macro_step(theta, rho, grad_theta, logp_theta, v, trajectory);
      // macro_step returns false on success, true on failure
      return result;
    }

    function build_span(span, v, depth, trajectory) {
      if (depth === 0) {
        // Macro step as leaf
        return build_leaf(span, v, trajectory);
      }
      var left = build_span(span, v, depth - 1, trajectory);
      if (!left) {
        return null;
      }
      var right = build_span(left, v, depth - 1, trajectory);
      if (!right) {
        return null;
      }
      if (uturn(left, right, v)) {
        return null;
      }
      // Barker update for subtree selection
      var combined = combine(left, right, true, v);
      return combined;
    }

    // transition
    var p0 = MultivariateNormal.getSample(self.dim);
    var q0 = self.chain.last().copy();
    var grad0 = self.gradLogDensity(q0);
    var logp0 = self.logDensity(q0) - p0.norm2() / 2;
    var span_accum = new Span(q0.copy(), p0.copy(), grad0.copy(), logp0, q0.copy(), p0.copy(), grad0.copy(), logp0, q0.copy(), logp0);
    for (var depth = 0; depth < 12; depth++) {
      var v = (Math.random() < 0.5) ? -1 : 1;
      var next_span = build_span(span_accum, v, depth, trajectory);
      if (!next_span) {
        break;
      }
      var combined_uturn = uturn(span_accum, next_span, v);
      // Metropolis update for selection (top-level)
      span_accum = combine(span_accum, next_span, false, v);
      if (combined_uturn) {
        break;
      }
    }

    var q = span_accum.theta_select_.copy();
    self.chain.push(q.copy());

    visualizer.queue.push({
      type: "proposal",
      proposal: q.copy(),
      nuts_trajectory: trajectory,
      initialMomentum: p0.copy(),
    });
    visualizer.queue.push({ type: "accept", proposal: q.copy() });
  },
});
