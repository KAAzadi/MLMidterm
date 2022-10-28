using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace MLMidterm
{
    public class Features
    {
        [LoadColumn(0)]
        public string id;

        [LoadColumn(1)]
        public string text;

        [LoadColumn(2)]
        public string text2;

        [LoadColumn(3), ColumnName("Labels")]
        public bool label;

        [LoadColumn(4), ColumnName("Features")]
        public double weights;
    }

    public class Test
    {
        [LoadColumn(0), ColumnName("ID")]
        public string id;

        [LoadColumn(1)]
        public string text;

        [LoadColumn(2)]
        public string text2;
    }

    public class Prediction : Test
    {

        [ColumnName("PredictedLabel")]
        public bool label { get; set; }
    }
}
