def bayes_theorem(prior_A, likelihood_B_given_A, marginal_B):
    """
    Compute the posterior probability P(A|B) using Bayes' Theorem.

    Parameters:
    - prior_A (float): Prior probability P(A), must be in [0, 1].
    - likelihood_B_given_A (float): Likelihood P(B|A), must be in [0, 1].
    - marginal_B (float): Marginal probability P(B), must be in (0, 1].

    Returns:
    - posterior (float): Posterior probability P(A|B).

    Raises:
    - ValueError: If input probabilities are outside their valid ranges.
    - ZeroDivisionError: If marginal_B is zero.
    """

    # Input validation
    if not (0 <= prior_A <= 1):
        raise ValueError(f"Invalid prior_A: {prior_A}. Must be between 0 and 1.")
    if not (0 <= likelihood_B_given_A <= 1):
        raise ValueError(f"Invalid likelihood_B_given_A: {likelihood_B_given_A}. Must be between 0 and 1.")
    if not (0 < marginal_B <= 1):
        raise ValueError(f"Invalid marginal_B: {marginal_B}. Must be between 0 and 1 (non-zero).")

    # Calculate posterior probability
    try:
        posterior = (likelihood_B_given_A * prior_A) / marginal_B
    except ZeroDivisionError as e:
        # This should not happen due to the input validation, but handle just in case
        raise ZeroDivisionError("Marginal probability P(B) is zero, cannot divide by zero.") from e

    # Ensure the result is a valid probability
    if not (0 <= posterior <= 1):
        # Due to floating-point arithmetic, result may be slightly outside [0, 1]
        posterior = max(0.0, min(1.0, posterior))

    return posterior

# Example usage
if __name__ == "__main__":
    # Example values
    prior_A = 0.02  # P(A): Prior probability of event A
    likelihood_B_given_A = 0.8  # P(B|A): Likelihood of B given A
    # Marginal probability P(B) can be calculated if unknown
    likelihood_B_given_not_A = 0.1  # P(B|¬A): Likelihood of B given not A
    prior_not_A = 1 - prior_A  # P(¬A)

    # Calculate marginal probability P(B)
    marginal_B = (likelihood_B_given_A * prior_A) + (likelihood_B_given_not_A * prior_not_A)

    # Compute posterior probability P(A|B)
    posterior = bayes_theorem(prior_A, likelihood_B_given_A, marginal_B)

    print(f"Posterior probability P(A|B): {posterior:.4f}")
