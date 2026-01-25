#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  int n;
  cin >> n;

  for (int i = 0; i < n; i++) {
    char *word = (char *)malloc(101 * sizeof(word));

    cin >> word;

    if (strlen(word) > 10) {
      cout << word[0] << strlen(word) - 2 << word[strlen(word) - 1] << endl;
    } else {
      cout << word << endl;
    }

    free(word);
  }

  return 0;
}
