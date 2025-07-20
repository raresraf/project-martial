#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  char c, word[1001];
  cin >> word;
  if (islower(word[0])) {
    c = toupper(word[0]);
    cout << c;
    for (int i = 1; i < strlen(word); i++)
      cout << word[i];
  } else
    cout << word;
  return 0;
}
