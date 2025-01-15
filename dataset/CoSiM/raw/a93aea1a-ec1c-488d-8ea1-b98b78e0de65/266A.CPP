#include <bits/stdc++.h>

using namespace std;

int main() {

  string s;
  int n;

  scanf("%d", &n);

  cin >> s;

  char last = s[0];
  int count = 0;

  for (int i = 1; i < n; ++i) {

    if (last == s[i])
      count++;
    else
      last = s[i];
  }

  cout << count << endl;
  ;
}