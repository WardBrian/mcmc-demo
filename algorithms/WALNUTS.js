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
    self.maxError = 0.1;
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

    function Span(theta_bk, rho_bk, grad_theta_bk, logp_bk, theta_fw, rho_fw, grad_theta_fw, logp_fw, theta_select, logp) {
      this.theta_bk_ = theta_bk;
      this.rho_bk_ = rho_bk;
      this.grad_theta_bk_ = grad_theta_bk;
      this.logp_bk_ = logp_bk;
      this.theta_fw_ = theta_fw;
      this.rho_fw_ = rho_fw;
      this.grad_theta_fw_ = grad_theta_fw;
      this.logp_fw_ = logp_fw;
      this.theta_select_ = theta_select;
      this.logp_ = logp;
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

    function within_tolerance(step, num_steps, theta_next, rho_next, grad_next, logp_next) {

      let half_step = 0.5 * step;
      let logp = logp_next;
      for (let n = 0; n < num_steps; ++n) {
        rho_next.increment(grad_next.scale(half_step));
        theta_next.increment(rho_next.scale(step));
        grad_next = self.gradLogDensity(theta_next);
        rho_next.increment(grad_next.scale(half_step));
      }

      let final_logp = self.logDensity(theta_next) + (-rho_next.norm2() / 2);
      return Math.abs(final_logp - logp) <= self.maxError;
    }

    function reversible(step, num_steps, theta, rho, grad, logp_next) {
      if (num_steps == 1) return true;
      let theta_next, rho_next, grad_next;
      while (num_steps >= 2) {
        theta_next = theta.copy();
        rho_next = rho.copy().scale(-1); // negate momentum
        grad_next = grad.copy();
        num_steps = Math.floor(num_steps / 2);
        step = step * 2;
        // If the backwards trajectory is within tolerance, it's not reversible
        if (within_tolerance(step, num_steps, theta_next, rho_next, grad_next, logp_next)) {
          return false;
        }
      }
      return true;
    }

    function macro_step(span, direction, trajectory) {
      const is_forward = (direction === 1);
      const theta = is_forward ? span.theta_fw_ : span.theta_bk_;
      const rho = is_forward ? span.rho_fw_ : span.rho_bk_;
      const grad = is_forward ? span.grad_theta_fw_ : span.grad_theta_bk_;
      const logp = is_forward ? span.logp_fw_ : span.logp_bk_;

      var step = is_forward ? self.dt : -self.dt;
      var theta_next, rho_next, grad_next, logp_next;

      for (var num_steps = 1, halvings = 0; halvings < 10; ++halvings, num_steps *= 2, step *= 0.5) {
        // Reset to initial state for this attempt
        theta_next = theta.copy();
        rho_next = rho.copy();
        grad_next = grad.copy();

        var leapfrogs = [];
        var half_step = 0.5 * step;

        // Leapfrog integration for num_steps
        for (var n = 0; n < num_steps; ++n) {
          rho_next.increment(grad_next.scale(half_step));
          theta_next.increment(rho_next.scale(step));
          grad_next = self.gradLogDensity(theta_next);
          rho_next.increment(grad_next.scale(half_step));
          leapfrogs.push({
            type: "leapfrog",
            halvings: halvings,
            from: (n === 0) ? theta.copy() : leapfrogs[leapfrogs.length - 1].to.copy(),
            to: theta_next.copy(),
            stepSize: Math.abs(step),
            step: n
          });
        }

        logp_next = self.logDensity(theta_next) + (-rho_next.norm2() / 2);

        // Check error tolerance first
        if (Math.abs(logp - logp_next) <= self.maxError) {
          // If within tolerance, check reversibility and return that result
          var is_reversible = reversible(step, num_steps, theta_next, rho_next, grad_next, logp_next);
          for (var lf of leapfrogs) trajectory.push(lf);
          trajectory.push({
            type: is_reversible ? "accept" : "reject",
            from: theta.copy(),
            to: theta_next.copy(),
          });
          // Return the result as an object to match C++ out parameters
          return {
            success: is_reversible,
            theta_next: theta_next,
            rho_next: rho_next,
            grad_next: grad_next,
            logp_next: logp_next
          };
        }
        // If not within tolerance, continue to next halving attempt
      }
      trajectory.push({
        type: "reject",
        from: theta.copy(),
        to: theta.copy(),
      });
      return { success: false }; // Failed after all halvings
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

    function combine(span_old, span_new, use_barker, direction) {
      var logp_old = span_old.logp_;
      var logp_new = span_new.logp_;
      var logp_total = log_sum_exp(logp_old, logp_new);
      var log_denominator;
      if (use_barker) {
        log_denominator = logp_total;
      } else { // Metropolis
        log_denominator = logp_old;
      }
      var update_logprob = logp_new - log_denominator;
      var update = Math.log(Math.random()) < update_logprob;
      var theta_select = update ? span_new.theta_select_ : span_old.theta_select_;

      var span_bk, span_fw;
      if (direction === 1) { // Forward
        span_bk = span_old;
        span_fw = span_new;
      } else { // Backward
        span_bk = span_new;
        span_fw = span_old;
      }
      return make_combined_span(
        span_bk, span_fw, theta_select, logp_total
      );
    }

    function build_leaf(span, direction, trajectory) {
      var result = macro_step(span, direction, trajectory);

      if (!result.success) {
        return null;
      }

      return make_leaf_span(result.theta_next, result.rho_next, result.grad_next, result.logp_next);
    }

    function build_span(span, direction, depth, trajectory) {
      if (depth === 0) {
        // Macro step as leaf
        return build_leaf(span, direction, trajectory);
      }
      var left = build_span(span, direction, depth - 1, trajectory);
      if (!left) {
        return null;
      }
      var right = build_span(left, direction, depth - 1, trajectory);
      if (!right) {
        return null;
      }
      if (uturn(left, right, direction)) {
        return null;
      }
      // Barker update for subtree selection
      var combined = combine(left, right, true, direction);
      return combined;
    }

    // transition
    var rho = MultivariateNormal.getSample(self.dim);
    var theta = self.chain.last().copy();
    var grad = self.gradLogDensity(theta);
    var logp = self.logDensity(theta) - rho.norm2() / 2;
    var span_accum = make_leaf_span(theta, rho, grad, logp);
    for (var depth = 0; depth < 12; depth++) {
      var direction = (Math.random() < 0.5) ? -1 : 1;
      trajectory.push({ type: direction > 0 ? "left" : "right" });

      var next_span = build_span(span_accum, direction, depth, trajectory);
      if (!next_span) {
        break;
      }
      var combined_uturn = uturn(span_accum, next_span, direction);
      // Metropolis update for selection (top-level)
      span_accum = combine(span_accum, next_span, false, direction);
      if (combined_uturn) {
        break;
      }
    }

    var theta_select = span_accum.theta_select_.copy();
    self.chain.push(theta_select.copy());

    visualizer.queue.push({
      type: "proposal",
      proposal: theta_select.copy(),
      nuts_trajectory: trajectory,
      initialMomentum: rho.copy(),
    });
    visualizer.queue.push({ type: "accept", proposal: theta_select.copy() });

    visualizer.draws += 1;
    document.getElementById("drawNum").innerHTML = "Draw " + visualizer.draws;

  },
});
