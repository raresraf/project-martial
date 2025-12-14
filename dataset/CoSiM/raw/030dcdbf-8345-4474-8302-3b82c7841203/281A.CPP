#include <iostream>
#include <string>
using namespace std;

int main() {
  string word;
  cin >> word;
  cout << static_cast<char>(toupper(word[0])) << word.substr(1, word.size())
       << endl;
  return 0;
}
