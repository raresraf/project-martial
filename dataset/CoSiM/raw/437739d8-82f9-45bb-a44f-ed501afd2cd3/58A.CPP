#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  string word = "hello";

  int aux = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == word[aux]) {
      aux++;
    }
  }

  if (aux == 5)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
