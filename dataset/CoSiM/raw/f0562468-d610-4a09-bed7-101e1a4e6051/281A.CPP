#include <iostream>
#include <string.h>
using namespace std;
int main() {
  string word;
  cin >> word;
  if (int(word[0]) >= 97) {
    word[0] = char(int(word[0]) - 32);
  }
  cout << word << endl;
  return 0;
}