# 自然语言处理-开放领域问答

对比常见模型在开放领域问答任务上的效果，主要涉及以下几种模型：

- [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/pdf/1704.00051v2.pdf)
- [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/pdf/1902.01718.pdf)
- [Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering](https://arxiv.org/pdf/1908.08167.pdf)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)

## Retriever

### DrQA

<table>
    <tr>
        <th></th>
        <th>Char</th>
        <th>Bigram</th>
        <th>Word</th>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.307</td>
        <td>0.361</td>
        <td><b>0.38</b></td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.201</td>
        <td>0.25</td>
        <td><b>0.256</b></td>
    </tr>
</table>

### BERTserini

<table>
    <tr>
        <th></th>
        <th>Char</th>
        <th>Bigram</th>
        <th>Word</th>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.314</td>
        <td><b>0.318</b></td>
        <td>0.281</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.426</td>
        <td><b>0.44</b></td>
        <td>0.433</td>
    </tr>
</table>

### Multi-passage BERT

<table>
    <tr>
        <th></th>
        <th>Char</th>
        <th>Bigram</th>
        <th>Word</th>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.598</td>
        <td><b>0.63</b></td>
        <td>0.622</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.578</td>
        <td><b>0.62</b></td>
        <td>0.61</td>
    </tr>
</table>


### DPR

<table>
    <tr>
        <th></th>
        <th>Faiss</th>
        <th>Faiss + ES Char</th>
        <th>Faiss + ES Bigram</th>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.821</td>
        <td><b>0.839</b></td>
        <td><b>0.839</b></td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.882</td>
        <td>0.896</td>
        <td><b>0.899</b></td>
    </tr>
</table>


## Reader 效果

### DrQA

<table>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="" colspan="4">Char</th>
        <th rowspan="" colspan="3">Word</th>
    </tr>
    <tr>
        <th>Embed Rand</th>
        <th>Embed Pretrained</th>
        <th>Embed Fixed</th>
        <th>Embed Rand + Bigram</th>
        <th>Embed Rand</th>
        <th>Embed Pretrained</th>
        <th>Embed Fixed</th>
    </tr>
    <tr>
        <td>dureader search</td>
        <td><b>0.671</b></td>
        <td>0.630</td>
        <td>0.622</td>
        <td>0.605</td>
        <td>0.566</td>
        <td>0.577</td>
        <td>0.57</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td><b>0.663</b></td>
        <td>0.627</td>
        <td>0.635</td>
        <td>0.614</td>
        <td>0.565</td>
        <td>0.574</td>
        <td>0.565</td>
    </tr>
</table>

### BERTserini

<table>
    <tr>
        <th></th>
        <th>Simple</th>
        <th>Fixed</th>
    </tr>
    <tr>
        <td>dureader search</td>
        <td><b>0.758</b></td>
        <td>0.429</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td><b>0.76</b></td>
        <td>0.45</td>
    </tr>
</table>

### Multi-passage BERT

<table>
    <tr>
        <th rowspan="2"></th>
        <th colspan="2">with negitive</th>
        <th colspan="2">without negitive</th>
    </tr>
    <tr>
        <th>Simple</th>
        <th>Fixed</th>
        <th>Simple</th>
        <th>Fixed</th>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.258</td>
        <td>0.004</td>
        <td><b>0.538</b></td>
        <td>0.261</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.248</td>
        <td>0.002</td>
        <td><b>0.486</b></td>
        <td>0.236</td>
    </tr>
</table>

## Combine 效果

### DrQA

<table>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="" colspan="3">Word Retriever + Char Reader</th>
    </tr>
    <tr>
        <td>Retriever Recall</td>
        <td>Reader F1</td>
        <td>Reader EM</td>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.38</td>
        <td>0.268</td>
        <td>0.141</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.256</td>
        <td>0.265</td>
        <td>0.139</td>
    </tr>
</table>

### BERTserini

<table>
    <tr>
        <td></td>
        <td>Retriever Recall</td>
        <td>Reader F1</td>
        <td>Reader EM</td>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.318</td>
        <td>0.267</td>
        <td>0.14</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.44</td>
        <td>0.322</td>
        <td>0.201</td>
    </tr>
</table>

### Multi-passage BERT

<table>
    <tr>
        <td></td>
        <td>Retriever Recall</td>
        <td>Reader F1</td>
        <td>Reader EM</td>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.452</td>
        <td>0.459</td>
        <td>0.303</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.441</td>
        <td>0.47</td>
        <td>0.32</td>
    </tr>
</table>


### DPR

<table>
    <tr>
        <td></td>
        <td>Retriever Recall</td>
        <td>Reader F1</td>
        <td>Reader EM</td>
    </tr>
    <tr>
        <td>dureader search</td>
        <td>0.511</td>
        <td>0.504</td>
        <td>0.367</td>
    </tr>
    <tr>
        <td>dureader zhidao</td>
        <td>0.672</td>
        <td>0.562</td>
        <td>0.438</td>
    </tr>
</table>
