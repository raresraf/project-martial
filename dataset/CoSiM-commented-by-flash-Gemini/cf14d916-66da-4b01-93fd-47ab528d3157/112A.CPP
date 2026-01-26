#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace std;
int main() {
  string word;
  string word2;

  cin >> word;
  cin >> word2;

  for (int s = 0; s < (int)word.length(); s++) {
    if (isupper(word[s]))
      word[s] = word[s] + 32;
    else
      ;
    if (isupper(word2[s]))
      word2[s] = word2[s] + 32;
    else
      ;
  }

  if (word == word2)
    cout << "0" << endl;

  else if (word < word2)
    cout << "-1" << endl;

  else if (word > word2)
    cout << "1" << endl;

  return 0;
}
