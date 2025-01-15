/**
 * I'm sorry I was an Idiot
 */
#include <iostream>
#include <stdio.h>

using namespace std;

int main() {
  int stops;
  scanf("%d", &stops);
  int val = 0;
  int max = 0;
  for (int i = 0; i < stops; ++i) {
    int out, in;
    cin >> out >> in;
    val -= out;
    val += in;
    if (val > max)
      max = val;
  }
  printf("%d\n", max);
  return 0;
}