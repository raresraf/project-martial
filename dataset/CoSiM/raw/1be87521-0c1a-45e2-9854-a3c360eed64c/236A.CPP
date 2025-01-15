#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  bool counts[26] = {};

  char ch;
  while (cin >> ch) {
    counts[ch - 'a'] = true;
  }

  if (accumulate(counts, counts + 26, 0) % 2 != 0) {
    cout << "IGNORE HIM!" << endl;
  } else {
    cout << "CHAT WITH HER!" << endl;
  }
}
