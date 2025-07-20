#include <bits/stdc++.h>

using namespace std;

int main() {

  string s;

  cin >> s;

  vector<int> v;

  for (int i = 0; i < s.length(); i = i + 2) {

    int k = s[i] - 48;

    v.push_back(k);
  }

  int z = s.length() / 2;

  sort(v.begin(), v.end());

  int j = 0;

  for (int i = 0; i < s.length(); i++) {

    if (i % 2 == 0) {

      cout << v[j];

      j++;

    }

    else {

      cout << '+';
    }
  }
}