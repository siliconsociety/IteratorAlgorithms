from setuptools import setup


with open("README.md", "r") as f:
    long_description = f.read()

dev_status = {
    "Planning": "Development Status :: 1 - Planning",
    "Pre": "Development Status :: 2 - Pre-Alpha",
    "Alpha": "Development Status :: 3 - Alpha",
    "Beta": "Development Status :: 4 - Beta",
    "Pro": "Development Status :: 5 - Production/Stable",
    "Mature": "Development Status :: 6 - Mature",
    "Inactive": "Development Status :: 7 - Inactive",
}

setup(
    name="IteratorAlgorithms",
    py_modules=["IteratorAlgorithms"],
    author="Robert Sharp",
    author_email="webmaster@sharpdesigndigital.com",
    version="0.1.5",
    description="A collection of iterator algorithms for Python3.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        dev_status["Beta"],
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        'accumulate', 'adjacent_difference', 'all_of', 'any_of', 'difference',
        'exclusive_scan', 'fork', 'generate', 'generate_n', 'inclusive_scan',
        'inner_product', 'intersection', 'iota', 'min_max', 'none_of',
        'partial_sum', 'partition', 'product', 'reduce', 'star_sum',
        'star_product', 'symmetric_difference', 'transform', 'transform_reduce',
        'transposed_sums', 'union', 'zip_transform',
    ],
    python_requires='>=3.6',
)
