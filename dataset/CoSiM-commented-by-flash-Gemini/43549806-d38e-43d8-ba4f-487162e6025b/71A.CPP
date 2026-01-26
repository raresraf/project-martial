#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, len = 0;
  char word[101];
  cin >> t;
  while (t--) {
    cin >> word;
    len = strlen(word);
    if (len <= 10)
      cout << word << '\n';
    else
      cout << word[0] << len - 2 << word[len - 1] << '\n';
  }
}
