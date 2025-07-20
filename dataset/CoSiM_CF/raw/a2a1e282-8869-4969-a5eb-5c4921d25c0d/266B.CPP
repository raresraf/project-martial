#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;

  string q;
  cin >> q;

  for (int i = 0; i < t; i++) {
    for (int j = 0; j < q.size() - 1; j++) {
      if (q.at(j) == 'B' && q.at(j + 1) == 'G') {
        q.at(j) = 'G';
        q.at(j + 1) = 'B';
        j++;
      }
    }
  }

  cout << q << endl;

  return 0;
}
