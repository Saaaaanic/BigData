using System.Diagnostics;

namespace MapReduce
{
    class Program
    {
        /// <summary>
        /// Standard method of computing the average of a list of numbers.
        /// Time Complexity: O(n)
        /// </summary>
        public static double StandardAverage(int[] array)
        {
            double total = 0;
            for (int i = 0; i < array.Length; i++)
            {
                total += array[i];
            }
            double average = total / array.Length;
            return average;
        }

        /// <summary>
        /// Computes the sum of a data chunk.
        /// </summary>
        public static double ChunkSum(int[] dataChunk)
        {
            double sum = 0;
            for (int i = 0; i < dataChunk.Length; i++)
            {
                sum += dataChunk[i];
            }
            return sum;
        }

        /// <summary>
        /// Computes the average of the given array using multiprocessing.
        /// Time Complexity: O(n/p + p), where p is the number of processes
        /// </summary>
        public static double MapReduceAverage(int[] array)
        {
            int numProcesses = Environment.ProcessorCount;
            int chunkSize = array.Length / numProcesses;
            double[] partialSums = new double[numProcesses];

            // Compute partial sums in parallel
            Parallel.For(0, numProcesses, i =>
            {
                int start = i * chunkSize;
                int end = (i == numProcesses - 1) ? array.Length : start + chunkSize;
                double sum = 0;
                for (int j = start; j < end; j++)
                {
                    sum += array[j];
                }
                partialSums[i] = sum;
            });

            // Sum all partial sums to get the total sum
            double total = partialSums.Sum();
            double average = total / array.Length;
            return average;
        }

        /// <summary>
        /// Generates an array of random integers.
        /// </summary>
        public static int[] GenerateRandomArray(int size)
        {
            Random rand = new Random();
            int[] array = new int[size];
            for (int i = 0; i < size; i++)
            {
                array[i] = rand.Next(0, 100);
            }
            return array;
        }

        static void Main(string[] args)
        {
            int[] testSizes = new int[] { 1000000, 10000000, 50000000, 100000000 };

            foreach (int arraySize in testSizes)
            {
                Console.WriteLine($"Testing with array size: {arraySize}");
                int[] array = GenerateRandomArray(arraySize);

                // Time StandardAverage
                Stopwatch sw = new Stopwatch();
                sw.Start();
                double standardAvg = StandardAverage(array);
                sw.Stop();
                double standardTime = sw.Elapsed.TotalMilliseconds;
                Console.WriteLine($"StandardAverage: {standardAvg}, Time: {standardTime} ms");

                // Time MapReduceAverage
                sw.Restart();
                double mapReduceAvg = MapReduceAverage(array);
                sw.Stop();
                double mapReduceTime = sw.Elapsed.TotalMilliseconds;
                Console.WriteLine($"MapReduceAverage: {mapReduceAvg}, Time: {mapReduceTime} ms");

                // Conclusion
                if (mapReduceTime < standardTime)
                {
                    Console.WriteLine("MapReduceAverage is faster than StandardAverage.");
                }
                else
                {
                    Console.WriteLine("StandardAverage is faster than MapReduceAverage.");
                }
                Console.WriteLine();
            }

            Console.WriteLine("Computation completed.");
        }
    }
}
