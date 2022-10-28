using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLMidterm;

class Program
{
    private static string filePath = Path.Combine(Environment.CurrentDirectory, "Data", "train_with_label.txt");
        
    static void Main(string[] args)
    {
        LRModel();
    }

    static void LRModel()
    {
        MLContext context = new MLContext();

        //Training Data
        Console.WriteLine("============TrainingSet==============");
        AddWeights("train_with_label_weighted.txt");
        TrainTestData train = LoadData(context,0.01f);
        ITransformer model = CreateModel(context, train.TrainSet);
        EvaluateModel(context, model, train.TestSet);

        Console.WriteLine("============DevSet==============");
        filePath = Path.Combine(Environment.CurrentDirectory, "Data", "dev_with_label.txt");
        AddWeights("dev_with_label_weighted.txt");
        TrainTestData dev = LoadData(context, 0.99f);
        EvaluateModel(context, model, dev.TestSet);


        Console.WriteLine("============TestSet==============");
        filePath = Path.Combine(Environment.CurrentDirectory, "Data", "test_without_label.txt");//ENTER FILE NAME OF GOLD LABEL HERE
        TrainTestData test = LoadTestData(context, 0.99f);
        EvaluateTest(context, model, test.TestSet);
    }

    static void AddWeights(string fileName)
    {
        string[] lines = File.ReadAllLines(filePath);

        for (int i = 0; i < lines.Length; i++)
        {
            //TRANSFORMATION - removing double quotes to better clean up data
            lines[i] = lines[i].ToLower();
            lines[i] = lines[i].Replace("\"", "");

            string[] temp = lines[i].Split('\t');
            double weight = (double)temp[2].Length / (double)temp[1].Length;

            int occurance = 0;
            foreach (string word in temp[1].Split(' '))
            {
                if (temp[2].Contains(word)) occurance++;
            }

            weight *= ((double)occurance / (double)temp[1].Length);

            //include weight of damerau-Levenshtein distance
            weight *= DamerauLevenshteinDistance(temp[1], temp[2]);

            lines[i] += "\t" + weight;
        }
        filePath = Path.Combine(Environment.CurrentDirectory, "Data", fileName);
        File.WriteAllLines(filePath, lines);
    }


    static TrainTestData LoadData(MLContext mlContext, float percent)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<Features>(filePath, allowQuoting: true, trimWhitespace: true);
        var preview = dataView.Preview();
        //create the data by using full file path
        //I was uncertain of any other methods rather than split, so I am splitting the training set to be 99% training, and dev set to be 0.1% training (latter not used for training)
        TrainTestData data = mlContext.Data.TrainTestSplit(dataView, testFraction: percent);

        return data;
    }

    static TrainTestData LoadTestData(MLContext mlContext, float percent)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<Test>(filePath, allowQuoting: true, trimWhitespace: true);
        var preview = dataView.Preview();
        //create the data by using full file path
        //I was uncertain of any other methods rather than split, so I am splitting the training set to be 99% training, and dev set to be 0.1% training (latter not used for training)
        TrainTestData data = mlContext.Data.TrainTestSplit(dataView, testFraction: percent);

        return data;
    }
    //Below 3 methods were created with help from https://programm.top/en/c-sharp/algorithm/damerau-levenshtein-distance/
    static int Minimum(int one, int two) => one < two ? one : two;

    static int Minimum(int one, int two, int three) => (one = one < two ? one : two) < three ? one : three;

    static int DamerauLevenshteinDistance(string firstText, string secondText)
    {
        var first = firstText.Length + 1;
        var second = secondText.Length + 1;
        var array = new int[first, second];

        for (var i = 0; i < first; i++)
        {
            array[i, 0] = i;
        }

        for (var j = 0; j < second; j++)
        {
            array[0, j] = j;
        }

        for (var i = 1; i < first; i++)
        {
            for (var j = 1; j < second; j++)
            {
                var cost = firstText[i - 1] == secondText[j - 1] ? 0 : 1;

                array[i, j] = Minimum(array[i - 1, j] + 1, // delete
                                                        array[i, j - 1] + 1, // insert
                                                        array[i - 1, j - 1] + cost); // replacement

                if (i > 1 && j > 1
                   && firstText[i - 1] == secondText[j - 2]
                   && firstText[i - 2] == secondText[j - 1])
                {
                    array[i, j] = Minimum(array[i, j],
                    array[i - 2, j - 2] + cost); // permutation
                }
            }
        }

        return array[first - 1, second - 1];
    }

    static ITransformer CreateModel(MLContext mlContext, IDataView set)
    {
        //data transformation
        var options = new TextFeaturizingEstimator.Options()
        {
            //set all to lower case
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,

            //feature to find ngrams throughout
            CharFeatureExtractor = new WordBagEstimator.Options()
            {
                NgramLength= 2,
                UseAllLengths = true
            },
        };

        //FeaturizeText only serves to set all text to lowercase and find ngrams, as seen above in options, as well as to convert all previously added features from AddWeights() into doubles.
        var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", options: options, inputColumnNames: new[] { nameof(Features.text), nameof(Features.text2)})
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Labels", featureColumnName: "Features"));
            

        var model = estimator.Fit(set);

        return model;
    }

    static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView set)
    {
        IDataView predictions = model.Transform(set);
        CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Labels");
        Console.WriteLine();
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    }

    static void EvaluateTest(MLContext mlContext, ITransformer model, IDataView set)
    {
        List<string> lines = new List<string>();
        IDataView predictions = model.Transform(set);
        Console.WriteLine();
        IEnumerable<Prediction> results = mlContext.Data.CreateEnumerable<Prediction>(predictions, reuseRowObject: false);
        foreach (var prediction in results)
        {
            lines.Add(prediction.id + "\t" + Convert.ToBoolean(prediction.label));
        }

        File.WriteAllLines("KianArmandMcCollumAzadi_test_result.txt", lines.ToArray());
    }
}
