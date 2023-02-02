import setuptools

long_description = """
    This is a library developed to incorporate useful properties and methods for predicting 
    creditworthiness of customers and expected loss for financial institutions. 
"""

setuptools.setup(
    name="",
    version="0.0.1",
    author="Gabriel Fernandes Luz",
    author_email="gfluz94@gmail.com",
    description="Package for credit risk assessment - PD, LGD, EAD, EL.",
    long_description=long_description,
    packages=list(
        filter(
            lambda x: x.startswith("credit_risk_modeling"),
            setuptools.find_packages(),
        )
    ),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        package.strip()
        for package in open("../requirements.txt", encoding="utf-8").readlines()
    ],
)
