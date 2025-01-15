#include <bits/stdc++.h>
using namespace std;

int main() {
  string word;
  getline(cin, word);
  transform(word.begin(), word.end(), word.begin(),
            ::tolower); // Converting uppercase to lowercase
  for (int i = 0; i < word.length();) {
    if (word[i] == 'a' || word[i] == 'e' || word[i] == 'i' || word[i] == 'o' ||
        word[i] == 'u' || word[i] == 'y') {
      word.erase(word.begin() + i); // Removing specific character
    } else {
      i++;
    }
  }
  for (char i : word) {
    cout << "." << i;
  }

  return 0;
}
