#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;

  cin >> n;

  /// Creating a 2D Vector & Taking Input
  vector<vector<bool>> matrix;
  for (int i = 0; i < n; i++) {
    vector<bool> known;
    for (int j = 0; j < 3; j++) {
      bool dec;
      cin >> dec;
      known.push_back(dec);
    }
    matrix.push_back(known);
  }

  /// Creating a vector to store all decision of all problem
  vector<int> inTotalDecision;
  for (int i = 0; i < matrix.size(); i++) {
    int sum = 0;
    for (int j = 0; j < matrix[i].size(); j++) {
      int temp;

      if (matrix[i][j] == false)
        temp = 0;
      else
        temp = 1;

      sum += temp;
    }
    if (sum >= 2)
      inTotalDecision.push_back(sum);
  }

  cout << inTotalDecision.size() << endl;
  return 0;
}

/**
3
1 1 0
1 1 1
1 0 0

2
1 0 0
0 1 1
**/
