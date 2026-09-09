/**
 * Plain-language guide for non-mathematical readers: what convex optimization
 * is, what the three problem types mean in real-world terms, what the four
 * methods do in everyday words, and one worked example the reader can
 * reproduce on this page. Written to the same honesty standard as the rest of
 * the UI: no promises the code does not keep.
 */
export default function ExplainerPanel() {
  return (
    <section className="panel explainer-panel" aria-labelledby="explainer-heading">
      <h2 id="explainer-heading">Plain-language guide: what is this actually doing?</h2>

      <p>
        Every button on this page runs the same simple idea. Imagine a hilly landscape where
        height means <em>how wrong the answer is</em>, and the valleys are the best answers. A
        solver starts somewhere and takes careful steps downhill until it reaches the bottom.
        The landscapes used here are <strong>convex</strong>, which is a guarantee worth having:
        they are shaped like a single bowl, with no hidden potholes or false bottoms. Wherever
        you start, careful downhill steps end at the <em>true</em> best answer, not a trap.
        That is the whole trick - and it is why convex optimization powers things from GPS to
        credit scoring.
      </p>

      <h3>The three problem types, in real-world terms</h3>
      <div className="explainer-grid">
        <div className="explainer-card">
          <h4>least_squares - "draw the best trend line"</h4>
          <p>
            You have measurements and want the straightest honest line through them. Real-world
            version: predict next month&apos;s rent from an apartment&apos;s size, age and
            location, using the last 80 rentals. The <em>condition</em> knob makes the problem
            harder or easier to walk: a high condition number is a long narrow valley instead of
            a round bowl, so the same careful steps take much longer to reach the bottom.
          </p>
        </div>
        <div className="explainer-card">
          <h4>lasso - "find the trend line that ignores junk"</h4>
          <p>
            You have many possible inputs, but most are noise. Real-world version: 50 sensors on
            a factory machine, but only a handful actually predict the failure. The{" "}
            <em>lam</em> knob sets how aggressively useless inputs get pushed to exactly zero.
            That is the special power here: the method <strong>selects</strong> the important
            inputs while it fits them, instead of needing a separate step.
          </p>
        </div>
        <div className="explainer-card">
          <h4>logistic - "answer yes/no questions"</h4>
          <p>
            Instead of predicting a number, sort things into two groups. Real-world version: will
            this customer cancel their subscription, given their usage and billing history? The
            answer is a probability between 0 and 1. The <em>ridge</em> knob keeps the model calm
            when the inputs overlap, so it does not overreact to one unusual customer.
          </p>
        </div>
      </div>

      <h3>The four methods, in everyday words</h3>
      <div className="explainer-grid">
        <div className="explainer-card">
          <h4>gd - "steady steps"</h4>
          <p>
            Look at the slope, take one carefully-sized step downhill, repeat. The step size is
            never tuned or guessed: it is derived from the problem itself, which is why the same
            run is exactly reproducible.
          </p>
        </div>
        <div className="explainer-card">
          <h4>nesterov - "steady steps with a glide"</h4>
          <p>
            Same idea, but each step carries momentum and looks slightly ahead, like a runner
            leaning into a curve. Mathematically guaranteed to reach the bottom in far fewer
            steps on the smooth problems - watch it in the iteration counts.
          </p>
        </div>
        <div className="explainer-card">
          <h4>ista - "steady steps that also clean house"</h4>
          <p>
            The lasso specialist. Each step both moves downhill and pushes truly useless inputs
            to exactly zero. Reliable and simple, one honest step at a time.
          </p>
        </div>
        <div className="explainer-card">
          <h4>fista - "clean house with a glide"</h4>
          <p>
            ista plus Nesterov-style momentum. Usually the fastest way to solve lasso, which is
            why the parser suggests it by default for lasso problems.
          </p>
        </div>
      </div>

      <h3>Try it yourself: a rent-prediction example</h3>
      <p>
        Suppose you are a property manager with data on <strong>80 apartments</strong> described
        by <strong>50 characteristics</strong> (size, floor, age, distance to transit, and so
        on), and you suspect only some characteristics really drive rent. You want the line that
        predicts rent <em>and</em> the shortlist of characteristics that matter.
      </p>
      <ol>
        <li>
          Choose problem <strong>lasso</strong> and method <strong>ista</strong> (or{" "}
          <strong>fista</strong> to see the accelerated variant converge in fewer steps).
        </li>
        <li>
          Fill in: <code>seed</code> 42, <code>n_rows</code> 80, <code>n_vars</code> 50,{" "}
          <code>lam</code> 0.5.
        </li>
        <li>Press Solve.</li>
      </ol>
      <p>
        The seed makes the synthetic dataset fully reproducible: anyone solving with the same
        numbers gets the identical answer, byte for byte. The page then shows the final{" "}
        <strong>objective</strong> (how wrong the best line is, lower is better), the{" "}
        <strong>gap</strong> shrinking toward zero in the chart (each step is making less and
        less difference - that is the walk settling into the bowl&apos;s bottom), and the{" "}
        <strong>ground truth</strong> this answer was checked against. The ground truth is
        computed independently with SciPy, a standard scientific library, so you are never asked
        to trust the from-scratch methods on faith.
      </p>
      <p className="truth-note">
        One honest state to know about: a 200 response with the &quot;hit the 2000-iteration
        cap&quot; badge means the walk stopped at its maximum number of steps before fully
        settling (big, awkwardly-shaped problems can do this). That is a real outcome shown
        truthfully, not an error - and the parameterized problems here are deliberately small
        enough (200 dimensions or fewer) that it stays rare.
      </p>
    </section>
  );
}
