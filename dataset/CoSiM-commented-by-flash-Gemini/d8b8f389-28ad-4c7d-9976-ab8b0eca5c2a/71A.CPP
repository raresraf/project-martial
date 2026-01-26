#include <iostream>
using namespace std;
int main() {
  int runs;
  cin >> runs;
  for (int run = 0; run < runs; ++run) {
    string word;
    cin >> word;
    if (word.size() <= 10)
      cout << word << endl;
    else
      cout << word[0] << (word.size() - 2) << word[word.size() - 1] << endl;
  }
}