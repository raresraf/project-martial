#include <iostream>

using namespace std;

int main() {
  string word;

  getline(cin, word);
  string::iterator it = word.begin();
  if (*it > 90)
    *it -= 32;
  cout << word << endl;

  return 0;
}