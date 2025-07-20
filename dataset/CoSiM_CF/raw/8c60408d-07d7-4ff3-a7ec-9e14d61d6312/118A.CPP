#include <bits/stdc++.h>
using namespace std;
int main() {
  string word, result;
  cin >> word;
  for (int i = 0; i < word.length(); i++) {
    if (word[i] == 'a' || word[i] == 'A' || word[i] == 'e' || word[i] == 'E' ||
        word[i] == 'i' || word[i] == 'I' || word[i] == 'o' || word[i] == 'O' ||
        word[i] == 'u' || word[i] == 'U' || word[i] == 'y' || word[i] == 'Y')
      continue;
    else {
      result += '.';
      result += tolower(word[i]);
    }
  }
  cout << result << endl;

  return 0;
}
