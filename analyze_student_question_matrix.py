# analyze_student_question_matrix.py

"""
This script analyzes student-question interaction data.
It computes various statistics including:
1. Student count
2. Question count
3. Matrix size
4. Sparsity
5. Distribution metrics
"""

import numpy as np
import pandas as pd

class StudentQuestionAnalyzer:
    def __init__(self, data):
        """Initialize with a DataFrame containing student-question interaction data."""
        self.data = data
        self.student_count = self.data['student_id'].nunique()
        self.question_count = self.data['question_id'].nunique()
        self.matrix_size = (self.student_count, self.question_count)

    def calculate_sparsity(self):
        """Calculate the sparsity of the interaction matrix."""
        total_entries = self.matrix_size[0] * self.matrix_size[1]
        non_zero_entries = self.data.shape[0]
        sparsity = 1 - (non_zero_entries / total_entries)
        return sparsity

    def get_distribution_metrics(self):
        """Get distribution metrics for questions answered by students."""
        distribution = self.data['question_id'].value_counts()
        return distribution.describe()

    def print_summary(self):
        """Prints a summary of the analysis."""
        print(f'Student Count: {self.student_count}')
        print(f'Question Count: {self.question_count}')
        print(f'Matrix Size: {self.matrix_size}')
        print(f'Sparsity: {self.calculate_sparsity()}')
        print(f'Distribution Metrics: {self.get_distribution_metrics()}')

# Example usage
if __name__ == '__main__':
    # Sample data creation (to be replaced by actual data)
    sample_data = {
        'student_id': [1, 2, 1, 2, 3],
        'question_id': [1, 1, 2, 3, 1]
    }
    df = pd.DataFrame(sample_data)
    analyzer = StudentQuestionAnalyzer(df)
    analyzer.print_summary()