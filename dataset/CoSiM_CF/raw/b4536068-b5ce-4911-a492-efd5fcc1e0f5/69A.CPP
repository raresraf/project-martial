#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  vector<vector<int>> matrix;

  cin >> n;

  for (int i = 0; i < n; i++) {
    vector<int> row;
    for (int i = 0; i < 3; i++) {
      int temp;
      cin >> temp;
      row.push_back(temp);
    }
    matrix.push_back(row);
  }

  for (int j = 0; j < 3; j++) {
    long int sum = 0;
    for (int i = 0; i < matrix.size(); i++) {
      sum += matrix[i][j];
    }
    if (sum != 0) {
      cout << "NO" << endl;
      return 0;
    }
  }
  cout << "YES" << endl;

  return 0;
}
