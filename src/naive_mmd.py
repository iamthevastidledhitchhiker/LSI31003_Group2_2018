def mmd(source, target):
	from math import exp, sqrt

	def sq(x):
		return x * x

	def average(xs):
		return sum(xs) / float(len(xs))

	def median(xs):
		return sorted(xs)[int(len(xs) / 2)]

	def product(x, y):
		return sum([ sq(x[i] - y[i]) for i in range(len(x)) ])

	def k(x, y):
		def gauss(var):
			return exp(product(x, y) / sq(float(var)))

		return sum(map(gauss, [p / 2.0, p, 2 * p]))

	def compute_p(xs):
		def get_dists(i):
			return [ sqrt(product(xs[i], xs[j])) for j in range(len(xs)) ]

	#	index 0 is the distance of the point with itself
		dists25 = [ average(sorted(get_dists(i))[1:26]) for i in range(len(xs)) ]
		return median(dists25)

#	param for the gaussian variances ; 'm' in the paper
	p = compute_p(target)

	n, m = float(len(source)), float(len(target))
	term1 = sum([ k(xi, xj) for xi in source for xj in source ])
	term2 = sum([ k(xi, yj) for xi in source for yj in target ])
	term3 = sum([ k(yi, yj) for yi in target for yj in target ])

	return term1 / sq(n) - 2 * term2 / (n * m) + term3 / sq(m)
