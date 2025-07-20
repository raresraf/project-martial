#include <iostream>
#include <string>
using namespace std;

int main() {

  int n;
  cin >> n;
  string word;
  for (int i = 0; i < n; i++) {
    cin >> word;

    if (word.length() <= 10)
      cout << word << endl;
    else {
      cout << word[0];
      cout << word.length() - 2;
      cout << word[word.length() - 1] << endl;
    }
  }

  return 0;
}
