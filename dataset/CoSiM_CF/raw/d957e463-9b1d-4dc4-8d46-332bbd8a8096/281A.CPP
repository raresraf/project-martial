#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string word;
  cin >> word;
  if (word[0] > 64 && word[0] < 91) {
    cout << word;
    return 0;
  } else {
    word[0] = toupper(word[0]);
    cout << word;
    return 0;
  }
}