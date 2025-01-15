#include <bits/stdc++.h>

using namespace std;

int main() {

  string s;
  string word = "hello";

  cin >> s;

  int j = 0, cont = 0;

  for (int i = 0; i < s.size(); i++) {
    if (s[i] == word[j]) {
      cont++;
      j++;
    }
  }

  if (cont == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
