# Julia Challenge

Repository to collect the code for:
https://nextjournal.com/sdanisch/the-julia-challenge


A submission should look the following:

    * place into language_name/authorname/solution.ext + benchmark.ext
    * a solution should implement an n-dimensional, n-argument lazy [broadcast](https://julia.guide/broadcasting) from scratch
    * lazy means, one can aggregate recursive calls to a broadcasting operation - and decide when and how to materialize the result
