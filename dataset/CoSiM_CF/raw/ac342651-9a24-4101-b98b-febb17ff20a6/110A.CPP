#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace std;

int main(void) {
  char input[20];
  int i;
  int output = 0;

  cin >> input;

  for (i = 0; input[i] != '\0'; i++)
    if (input[i] == '4' || input[i] == '7') {
      output++;
    }

  if (output == 4 || output == 7)
    cout << "YES";
  else
    cout << "NO";
}