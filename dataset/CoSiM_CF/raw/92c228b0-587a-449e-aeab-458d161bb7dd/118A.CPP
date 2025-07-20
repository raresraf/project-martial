#include <bits/stdc++.h>
using namespace std;
int main() {
  string word, sa, ca, sca;
  sa = "abcdefghijklmnopqrstuvwxyz";
  ca = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  sca = "aoyeiu";
  cin >> word;
  for (int i = 0; i < word.length(); i++) {
    for (int j = 0; j < 26; j++) {
      if (word[i] == ca[j]) {
        word[i] = sa[j];
      }
    }
  }
  for (int k = 0; k < word.length(); k++) {
    for (int f = 0; f < 6; f++) {
      if (word[k] == sca[f]) {
        word.erase(k, 1);
        k--;
      }
    }
  }
  for (int b = 0; b < word.length(); b++)
    cout << "." << word[b];
  return 0;
}
