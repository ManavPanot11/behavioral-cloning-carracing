"""
Stage 0A – Toy gradient descent
Reproduces the article's hand-checkable example.

Picture:
  [[8, 8, 1],
   [8, 8, 1],
   [8, 8, 2]]

x = (right column total) - (left column total) = (1+1+2) - (8+8+8) = 4 - 24 = -20
label = -15
w starts at -1, learning rate = 0.001

Update rule:
  guess = w * x
  miss  = guess - label
  slope = 2 * miss * x
  w     = w - lr * slope
"""

print("=" * 60)
print("Toy Gradient Descent – Hand-Checkable Example")
print("=" * 60)

# Setup
picture = [[8, 8, 1], [8, 8, 1], [8, 8, 2]]
left_col_total = sum(row[0] for row in picture)
right_col_total = sum(row[2] for row in picture)
x = right_col_total - left_col_total
print(f"Left col total = {left_col_total}, Right col total = {right_col_total}")
print(f"x = {right_col_total} - {left_col_total} = {x}")

label = -15
w = -1.0
lr = 0.001

# Expected values
expected_w = [0.4, 0.68, 0.736]
expected_guess = [-8.0, -13.6, -14.72]
expected_slopes_start_and_rounds = [-1400, -280, -56, -11.2]

print(f"\nlabel = {label}, w_init = {w}, lr = {lr}")
print(f"x = {x}\n")

header = f"{'Round':<8}{'w':<12}{'guess':<12}{'miss':<12}{'slope':<12}"
print(header)
print("-" * len(header))

# Start (round 0): compute guess, miss, slope BEFORE any update
guess = w * x
miss = guess - label
slope = 2 * miss * x
print(f"{'start':<8}{w:<12.4f}{guess:<12.4f}{miss:<12.4f}{slope:<12.4f}")

assert abs(slope - expected_slopes_start_and_rounds[0]) < 1e-6, \
    f"Start slope mismatch: {slope} != {expected_slopes_start_and_rounds[0]}"

for i in range(1, 4):
    w = w - lr * slope
    guess = w * x
    miss = guess - label
    new_slope = 2 * miss * x
    print(f"{i:<8}{w:<12.4f}{guess:<12.4f}{miss:<12.4f}{new_slope:<12.4f}")

    assert abs(w - expected_w[i - 1]) < 1e-6, \
        f"Round {i} w mismatch: {w} != {expected_w[i - 1]}"
    assert abs(guess - expected_guess[i - 1]) < 1e-6, \
        f"Round {i} guess mismatch: {guess} != {expected_guess[i - 1]}"
    assert abs(new_slope - expected_slopes_start_and_rounds[i]) < 1e-6, \
        f"Round {i} slope mismatch: {new_slope} != {expected_slopes_start_and_rounds[i]}"

    slope = new_slope

print("\nAll assertions passed!")
