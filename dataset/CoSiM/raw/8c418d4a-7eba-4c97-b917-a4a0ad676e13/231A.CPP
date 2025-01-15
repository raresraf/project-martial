// http://codeforces.com/contest/231/problem/A

#include <iostream>
#include <vector>
using namespace std;

int main() {
  int n;
  cin >> n;

  vector<vector<int>> solutions(n, vector<int>(3));
  for (short row = 0; row < n; row++) {
    cin >> solutions[row][0] >> solutions[row][1] >> solutions[row][2];
  }

  int accepted{};
  for (short row = 0; row < n; row++) {
    if (solutions[row][0] + solutions[row][1] + solutions[row][2] >= 2) {
      accepted++;
    }
  }

  cout << accepted;
}