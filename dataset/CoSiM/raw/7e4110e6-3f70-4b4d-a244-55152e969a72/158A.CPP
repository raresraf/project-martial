#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, counter = 0;

  cin >> n >> k;
  vector<int> marks;
  for (int i = 1; i <= n; i++) {
    int a;
    cin >> a;
    marks.push_back(a);
  }

  for (int it = 0; it < marks.size(); it++) {
    if ((marks[it] >= marks[k - 1]) && (marks[it] != 0))
      counter++;
  }
  cout << counter << endl;

  return 0;
}

/**
8 5
10 9 8 7 7 7 5 5
output:
6

4 2
0 0 0 0
output:
0

5 5
1 1 1 1 1
output:
5
**/
