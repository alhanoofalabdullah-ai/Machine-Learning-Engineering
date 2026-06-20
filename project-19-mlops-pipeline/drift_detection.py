from scipy.stats import ks_2samp

reference = [1,2,3,4,5]
current = [2,3,4,5,6]

stat, p = ks_2samp(
    reference,
    current
)

print(
    "P-value:",
    p
)
