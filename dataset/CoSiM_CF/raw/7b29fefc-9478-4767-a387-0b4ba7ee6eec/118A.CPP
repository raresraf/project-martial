// Solu  o incompleta

#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  string word;
  cin >> word;

  char vogals[] = "AEIOUaeiou";
  for (uint i = 0; i < strlen(vogals); i++)
    word.erase(remove(word.begin(), word.end(), vogals[i]), word.end());

  transform(word.begin(), word.end(), word.begin(), ::tolower);

  string result;

  for (uint i = 0; i < word.length(); i++) {
    result.append(".");
    result += word[i];
  }

  cout << result << endl;

  return 0;
}
