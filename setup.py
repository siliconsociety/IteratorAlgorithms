from setuptools import setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="IteratorAlgorithms",
    py_modules=["IteratorAlgorithms"],
    author="Robert Sharp",
    author_email="webmaster@sharpdesigndigital.com",
    version="0.0.2",
    description="A collection of iterator algorithms for Python3.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        'accumulate', 'adjacent_difference', 'all_of', 'analytic_continuation',
        'any_of', 'difference', 'exclusive_scan', 'flatten', 'fork', 'generate',
        'generate_n', 'inclusive_scan', 'inner_product', 'intersection', 'iota',
        'min_max', 'none_of', 'partial_sum', 'partition', 'product', 'reduce',
        'symmetric_difference', 'transform', 'transform_reduce',
        'transposed_sums', 'union', 'zip_transform',
    ],
    python_requires='>=3.6',
)
