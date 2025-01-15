
#include <iostream>
#include <string>
using namespace std;

int main() {
  string word;
  auto test = 0;
  auto n = 0;

  cin >> test;

  while (test--) {
    cin >> word;

    if (word == "X++")
      n++;
    else if (word == "X--")
      n--;
    else if (word == "++X")
      ++n;
    else if (word == "--X")
      --n;
  }

  cout << n << endl;
}