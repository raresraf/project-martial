#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  string line;
  cin >> line;

  int i = 1;
  int n = line.size();
  const int minDanger = 7;
  bool danger = false;
  while (i < n) {
    int same = 1;
    while (i < n && line[i - 1] == line[i]) {
      ++i;
      ++same;
    }

    if (same >= minDanger)
      danger = true;

    ++i;
  }

  if (danger)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}