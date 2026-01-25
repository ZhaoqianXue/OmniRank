# OmniRank: A Large-Language-Model Agent Platform for Statistically Rigorous Ranking Inference from Arbitrary Multiway Comparisons

## Target Journal: Journal of the American Statistical Association - Applications and Case Studies

## Background
`docs/literature/spectral_ranking_inferences.md` 的作者 Mengxin Yu 提出了一种基于谱理论的排序推断方法，可以处理任意多向比较数据，并且具有统计最优性。目前该方法主要以R包的形式存在，用户需要具备一定的统计学背景/编程能力才能使用。

## Objectives
Mengxin Yu 希望我可以借鉴几篇已经发表在顶级期刊上的关于大语言模型智能体平台的文章（因为这些文章已经发表在顶级期刊上，证明了这几篇文章中LLM Agent的架构是可行的），然后我们希望构建一个基于大语言模型的智能体平台，将该方法封装成一个用户友好的工具，让非统计学背景的用户也能方便地使用该方法进行排序推断。

## Reviewer Concerns
目标期刊的审稿人可能会对 OmniRank 的架构提出质疑，因此我们需要参考已经发表在顶级期刊上的关于大语言模型智能体的文章，从而避免 LLM Agent 在 OmniRank 中参与感过于薄弱，以至于被 reviewer 质疑 OmniRank 只是简单地调用了 API（LLM Agent as a Wrapper）。

## Construction Strategy
OmniRank 将采取 top-down 的构建思路，即先完成文章的写作，使其与已经发表在顶级期刊上的关于大语言模型智能体的文章处于同一水平，达到顶级期刊的标准，然后再实现 OmniRank 的代码。

## Already Published Articles
1. `docs/literature/automated_hypothesis_validation.md`
2. `docs/literature/clinical_prediction_models.md`
3. `docs/literature/lambda.md`
4. `docs/literature/tissuelab.md`
尤其要参考 `docs/literature/lambda.md`，（指写作风格，写作结构，不参考具体内容）因为该文章发表在 Journal of the American Statistical Association - Applications and Case Studies 上，说明该文章的写作质量是符合顶级期刊要求的。