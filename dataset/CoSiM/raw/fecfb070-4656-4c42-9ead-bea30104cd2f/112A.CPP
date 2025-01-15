#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;

int main() {
  char arr1[101], arr2[101];
  scanf("%s", &arr1);
  scanf("%s", &arr2);
  int len = strlen(arr1);
  int i;
  for (i = 0; i < len; i++) {
    if (arr1[i] >= 97 && arr1[i] <= 122) {
      arr1[i] = 'A' + (arr1[i] - 'a');
    }
    if (arr2[i] >= 97 && arr2[i] <= 122) {
      arr2[i] = 'A' + (arr2[i] - 'a');
    }
  }
  for (i = 0; i < len; i++) {
    if (arr1[i] < arr2[i]) {
      printf("-1\n");
      break;
    } else if (arr1[i] > arr2[i]) {
      printf("1\n");
      break;
    } else if (i == len - 1) {
      printf("0\n");
      break;
    }
  }
  return 0;
}
