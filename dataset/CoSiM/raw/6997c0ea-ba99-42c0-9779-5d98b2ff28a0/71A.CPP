#include <iostream>
#include <string>

using namespace std;

int main() {
  string word;
  int num;
  cin >> num;
  for (int i = 0; i < num; i++) {
    cin >> word;
    if (word.length() > 10) {
      cout << word.at(0) << word.length() - 2 << word.at(word.length() - 1)
           << endl;
    } else {
      cout << word << endl;
    }
  }
}
