#include <bits/stdc++.h>

using namespace std;

int main() {
  string word;
  cin >> word;
  if (islower(word[0])) {
    word[0] = word[0] - 32;
  }
  cout << word;

  return 0;
}
