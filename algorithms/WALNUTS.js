"use strict";

// based on EfficentNUTS.js
// and Bob Carpenter's C++ implementation.
// Changes are localized to the j==0 case, replacing
// the leapfrog step with a call to the WALNUTS macro step.

MCMC.registerAlgorithm("WALNUTS", {
  description: "Sub-orbit adapation for the No-U-Turn Sampler",

  about: function () {
    // TODO: add real link
    window.open("https://github.com/bob-carpenter/walnuts");
  },

  init: function (self) {
    self.dt = 0.6;
    self.maxError = 0.8;
  },

  reset: function (self) {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: function (self, folder) {
    folder.add(self, "dt", 0.025, 2.5).step(0.025).name("Leapfrog &Delta;t");
    folder.add(self, "maxError", 0.01, 1).step(0.01).name("Max error");
    folder.open();
  },

  step: function (self, visualizer) {
    var trajectory = [];

    // Add this helper function at the top level of the file (inside or outside the MCMC.registerAlgorithm block)
    function within_tolerance(step, numSteps, q, p, grad, logp) {
      let q_next = q.copy();
      let p_next = p.copy();
      let grad_next = grad.copy();
      let logp_next = logp;
      let halfStep = 0.5 * step;
      let logp_min = logp_next;
      let logp_max = logp_next;
      for (let n = 0; n < numSteps; ++n) {
        p_next.increment(grad_next.scale(halfStep));
        q_next.increment(p_next.scale(step));
        grad_next = self.gradLogDensity(q_next);
        p_next.increment(grad_next.scale(halfStep));
        logp_next = self.logDensity(q_next) - p_next.norm2() / 2;
        logp_min = Math.min(logp_min, logp_next);
        logp_max = Math.max(logp_max, logp_next);
        if (logp_max - logp_min > self.maxError) {
          return false;
        }
      }
      return true;
    }

    function reversible(step, numSteps, q, p, grad, logp) {
      if (numSteps < 2) return false; // base case: single step is always reversible
      let q_next = q.copy();
      let p_next = p.copy().scale(-1); // negate momentum
      let grad_next = grad.copy();
      let logp_next = logp;
      while (numSteps >= 2) {
        q_next = q.copy();
        p_next = p.copy().scale(-1);
        grad_next = grad.copy();
        let nSteps = Math.floor(numSteps / 2);
        let bigStep = step * 2;
        // Use within_tolerance for the error check
        if (within_tolerance(bigStep, nSteps, q_next, p_next, grad_next, logp_next)) {
          return false; // reversible
        }
        numSteps = nSteps;
        step = bigStep;
      }
      return true; // not reversible if loop completes
    }

    // Extracted WALNUTS macro step with error control and adaptive step size
    function walnutsMacroStep(q, p, u, v, trajectory) {
      var q0 = q.copy();
      var p0 = p.copy();
      var grad0 = self.gradLogDensity(q0);
      var logp0 = self.logDensity(q0) - p0.norm2() / 2;
      var maxHalvings = 10;
      var found = false;
      var q1, p1, grad1, logp1, logp_min, logp_max;
      var finalLeapfrogs = [];
      var origStep = Math.abs(self.dt);
      var direction = v;
      for (var halvings = 0, numSteps = 1, stepSize = direction * origStep; halvings < maxHalvings; ++halvings, numSteps *= 2, stepSize *= 0.5) {
        q1 = q0.copy();
        p1 = p0.copy();
        grad1 = grad0.copy();
        logp1 = logp0;
        logp_min = logp0;
        logp_max = logp0;
        var leapfrogs = [];
        var halfStep = stepSize / 2;
        for (var n = 0; n < numSteps; ++n) {
          p1.increment(grad1.scale(halfStep));
          q1.increment(p1.scale(stepSize));
          grad1 = self.gradLogDensity(q1);
          p1.increment(grad1.scale(halfStep));
          logp1 = self.logDensity(q1) - p1.norm2() / 2;
          logp_min = Math.min(logp_min, logp1);
          logp_max = Math.max(logp_max, logp1);
          leapfrogs.push({
            type: "leapfrog",
            halvings: halvings,
            from: (n === 0) ? q0.copy() : leapfrogs[leapfrogs.length - 1].to,
            to: q1.copy(),
            stepSize: stepSize,
            step: n
          });
          if (logp_max - logp_min > self.maxError) break;
        }
        if (logp_max - logp_min <= self.maxError) {
          if (!reversible(stepSize, numSteps, q0, p0, grad0, logp0)) {
            found = true;
            finalLeapfrogs = leapfrogs;
            break;
          }
        }
      }
      var n_ = (found && u < Math.exp(logp1)) ? 1 : 0;
      var s_ = found ? 1 : 0;
      for (var lf of finalLeapfrogs) {
        trajectory.push(lf);
      }
      trajectory.push({
        type: n_ == 1 && s_ == 1 ? "accept" : "reject",
        from: q0.copy(),
        to: q1.copy(),
      });
      return { q_p: q1, p_p: p1, q_m: q1, p_m: p1, q_: q1, n_: n_, s_: s_ };
    }

    // BuildTree from Algorithm 3: Efficient No-U-Turn Sampler
    function buildTree(q, p, u, v, j) {
      var q = q.copy();
      if (j == 0) {
        return walnutsMacroStep(q, p, u, v, trajectory);
      } else {
        // recursion - build the left and right subtrees
        var result = buildTree(q, p, u, v, j - 1);
        var q_m = result.q_m,
          p_m = result.p_m,
          q_p = result.q_p,
          p_p = result.p_p,
          q_ = result.q_,
          n_ = result.n_,
          s_ = result.s_;
        if (s_ == 1) {
          var n__, s__, q__;
          if (v == -1) {
            var result = buildTree(q_m, p_m, u, v, j - 1);
            q_m = result.q_m;
            p_m = result.p_m;
            q__ = result.q_;
            n__ = result.n_;
            s__ = result.s_;
          } else {
            var result = buildTree(q_p, p_p, u, v, j - 1);
            q_p = result.q_p;
            p_p = result.p_p;
            q__ = result.q_;
            n__ = result.n_;
            s__ = result.s_;
          }
          if (Math.random() < n__ / (n_ + n__)) q_ = q__;
          s_ = s_ * s__ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
          n_ = n_ + n__;
        }
        return {
          q_p: q_p,
          p_p: p_p,
          q_m: q_m,
          p_m: p_m,
          q_: q_,
          n_: n_,
          s_: s_,
        };
      }
    }

    var p0 = MultivariateNormal.getSample(self.dim);
    var u = Math.random() * Math.exp(self.logDensity(self.chain.last()) - p0.norm2() / 2);

    var q = self.chain.last().copy(),
      q_m = self.chain.last().copy(),
      q_p = self.chain.last().copy(),
      p_m = p0.copy(),
      p_p = p0.copy(),
      j = 0,
      n = 1,
      s = 1;

    while (s == 1) {
      var v = Math.sign(Math.random() - 0.5);
      var q_, n_, s_;
      if (v == -1) {
        var result = buildTree(q_m, p_m, u, v, j);
        q_m = result.q_m;
        p_m = result.p_m;
        q_ = result.q_;
        n_ = result.n_;
        s_ = result.s_;
      } else {
        var result = buildTree(q_p, p_p, u, v, j);
        q_p = result.q_p;
        p_p = result.p_p;
        q_ = result.q_;
        n_ = result.n_;
        s_ = result.s_;
      }
      if (s_ == 1 && Math.random() < n_ / n) q = q_.copy();
      s = s_ * (q_p.subtract(q_m).dot(p_m) >= 0 ? 1 : 0) * (q_p.subtract(q_m).dot(p_p) >= 0 ? 1 : 0);
      n = n + n_;
      j = j + 1;
    }

    self.chain.push(q.copy());

    visualizer.queue.push({
      type: "proposal",
      proposal: q,
      nuts_trajectory: trajectory,
      initialMomentum: p0,
    });
    visualizer.queue.push({ type: "accept", proposal: q });
  },
});
