// C++ program to sort a string of characters
#include <bits/stdc++.h>
using namespace std;

int main() {
  char s[101];
  int i, n, counter = 0;
  cin >> s;
  n = strlen(s);
  sort(s, s + n);

  for (i = 0; i < n; i++) {
    if (s[i] != s[i + 1])
      counter++;
  }

  if (counter % 2 == 1)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}
