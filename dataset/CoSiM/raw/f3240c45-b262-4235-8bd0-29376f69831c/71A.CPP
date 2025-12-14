#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  int n;
  cin >> n;
  vector<string> words;
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;
    words.push_back(word);
  }

  for (string &word : words) {
    int s = word.size();
    if (s <= 10) {
      cout << word << endl;
    } else {
      int val = s - 2;
      cout << word[0] << val << word[s - 1] << endl;
    }
  }

  return 0;
}