#include <iostream>
using namespace std;
int main() {
  int x;
  cin >> x;
  string word;
  while (x > 0) {
    cin >> word;
    if (word.size() > 10) {
      cout << word[0] << word.size() - 2 << word[word.size() - 1] << endl;
    } else
      cout << word << endl;

    x--;
  }
  return 0;
}
