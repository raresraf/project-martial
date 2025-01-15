#include <cctype>
#include <iostream>

using namespace std;

int main() {

  string word;
  cin >> word;

  if (islower(word.at(0))) {
    word.at(0) -= 32;
  }

  cout << word << endl;

  return 0;
}