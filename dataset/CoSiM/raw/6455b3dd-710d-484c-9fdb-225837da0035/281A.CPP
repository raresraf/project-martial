#include <iostream>
using namespace std;
int main() {
  string word;
  cin >> word;
  if (word[0] >= 97 && word[0] <= 122) {
    word[0] -= 32;
  }
  cout << word;
}
