#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace std;
int main() {
  string word;

  cin >> word;

  if (isupper(word[0]))
    ;
  else
    word[0] = word[0] - 32;

  cout << word << endl;

  return 0;
}
