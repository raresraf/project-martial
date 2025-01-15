#include <stdio.h>

int main(void) {

  //
  //    : 1 49 / 2 50 / 3 51
  //     : + 43

  //      1, 3, 5,...      .
  //     0, 2, 4, 6...      .

  char s[100];
  int cnt1 = 0;
  int cnt2 = 0;
  int cnt3 = 0;
  int idx = 0;

  scanf("%s", s);

  for (int i = 0; s[i] != '\0'; i = i + 2) {
    if (s[i] == 49) {
      cnt1++;
    }
    if (s[i] == 50) {
      cnt2++;
    }
    if (s[i] == 51) {
      cnt3++;
    }
  }

  if (cnt1 + cnt2 + cnt3 == 1) {
    printf("%s", s);
  } else {
    while (cnt1 > 0) {
      s[idx] = '1';
      idx++;
      cnt1--;
      if (cnt1 + cnt2 + cnt3 != 0) {
        s[idx] = '+';
        idx++;
      }
    }
    while (cnt2 > 0) {
      s[idx] = '2';
      idx++;
      cnt2--;
      if (cnt1 + cnt2 + cnt3 != 0) {
        s[idx] = '+';
        idx++;
      }
    }
    while (cnt3 > 0) {
      s[idx] = '3';
      idx++;
      cnt3--;
      if (cnt1 + cnt2 + cnt3 != 0) {
        s[idx] = '+';
        idx++;
      }
    }
    printf("%s", s);
  }

  return 0;
}
