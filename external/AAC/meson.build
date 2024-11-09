# meson.build of the external library "AAC"
# David Weenink, 9 November 2024

AACDEC_SRC = '''
	libAACdec/src/FDK_delay.cpp
	libAACdec/src/aac_ram.cpp
	libAACdec/src/aac_rom.cpp
	libAACdec/src/aacdec_drc.cpp
	libAACdec/src/aacdec_hcr.cpp
	libAACdec/src/aacdec_hcr_bit.cpp
	libAACdec/src/aacdec_hcrs.cpp
	libAACdec/src/aacdec_pns.cpp
	libAACdec/src/aacdec_tns.cpp
	libAACdec/src/aacdecoder.cpp
	libAACdec/src/aacdecoder_lib.cpp
	libAACdec/src/block.cpp
	libAACdec/src/channel.cpp
	libAACdec/src/channelinfo.cpp
	libAACdec/src/conceal.cpp
	libAACdec/src/ldfiltbank.cpp
	libAACdec/src/pulsedata.cpp
	libAACdec/src/rvlc.cpp
	libAACdec/src/rvlcbit.cpp
	libAACdec/src/rvlcconceal.cpp
	libAACdec/src/stereo.cpp
	libAACdec/src/usacdec_ace_d4t64.cpp
	libAACdec/src/usacdec_ace_ltp.cpp
	libAACdec/src/usacdec_acelp.cpp
	libAACdec/src/usacdec_fac.cpp
	libAACdec/src/usacdec_lpc.cpp
	libAACdec/src/usacdec_lpd.cpp
	libAACdec/src/usacdec_rom.cpp
	libAACdec/src/aac_ram.h
	libAACdec/src/aac_rom.h
	libAACdec/src/aacdec_drc.h
	libAACdec/src/aacdec_drc_types.h
	libAACdec/src/aacdec_hcr.h
	libAACdec/src/aacdec_hcr_bit.h
	libAACdec/src/aacdec_hcr_types.h
	libAACdec/src/aacdec_hcrs.h
	libAACdec/src/aacdec_pns.h
	libAACdec/src/aacdec_tns.h
	libAACdec/src/aacdecoder.h
	libAACdec/src/block.h
	libAACdec/src/channel.h
	libAACdec/src/channelinfo.h
	libAACdec/src/conceal.h
	libAACdec/src/conceal_types.h
	libAACdec/src/FDK_delay.h
	libAACdec/src/ldfiltbank.h
	libAACdec/src/overlapadd.h
	libAACdec/src/pulsedata.h
	libAACdec/src/rvlc.h
	libAACdec/src/rvlc_info.h
	libAACdec/src/rvlcbit.h
	libAACdec/src/rvlcconceal.h
	libAACdec/src/stereo.h
	libAACdec/src/usacdec_ace_d4t64.h
	libAACdec/src/usacdec_ace_ltp.h
	libAACdec/src/usacdec_acelp.h
	libAACdec/src/usacdec_const.h
	libAACdec/src/usacdec_fac.h
	libAACdec/src/usacdec_lpc.h
	libAACdec/src/usacdec_lpd.h
	libAACdec/src/usacdec_rom.h'''.split()
	
ARITHCODING_SRC = [ 'libArithCoding/src/ac_arith_coder.cpp' ]
	
DRCDEC_SRC = '''
	libDRCdec/src/FDK_drcDecLib.cpp
	libDRCdec/src/drcDec_gainDecoder.cpp
	libDRCdec/src/drcDec_reader.cpp
	libDRCdec/src/drcDec_rom.cpp
	libDRCdec/src/drcDec_selectionProcess.cpp
	libDRCdec/src/drcDec_tools.cpp
	libDRCdec/src/drcGainDec_init.cpp
	libDRCdec/src/drcGainDec_preprocess.cpp
	libDRCdec/src/drcGainDec_process.cpp
	libDRCdec/src/drcDec_gainDecoder.h
	libDRCdec/src/drcDec_reader.h
	libDRCdec/src/drcDec_rom.h
	libDRCdec/src/drcDec_selectionProcess.h
	libDRCdec/src/drcDec_tools.h
	libDRCdec/src/drcDec_types.h
	libDRCdec/src/drcDecoder.h
	libDRCdec/src/drcGainDec_init.h
	libDRCdec/src/drcGainDec_preprocess.h
	libDRCdec/src/drcGainDec_process.h'''.split()
	
FDK_SRC = '''
  libFDK/src/FDK_bitbuffer.cpp
  libFDK/src/FDK_core.cpp
  libFDK/src/FDK_crc.cpp
  libFDK/src/FDK_decorrelate.cpp
  libFDK/src/FDK_hybrid.cpp
  libFDK/src/FDK_lpc.cpp
  libFDK/src/FDK_matrixCalloc.cpp
  libFDK/src/FDK_qmf_domain.cpp
  libFDK/src/FDK_tools_rom.cpp
  libFDK/src/FDK_trigFcts.cpp
  libFDK/src/autocorr2nd.cpp
  libFDK/src/dct.cpp
  libFDK/src/fft.cpp
  libFDK/src/fft_rad2.cpp
  libFDK/src/fixpoint_math.cpp
  libFDK/src/huff_nodes.cpp
  libFDK/src/mdct.cpp
  libFDK/src/nlc_dec.cpp
  libFDK/src/qmf.cpp
  libFDK/src/scale.cpp
  '''.split()

MPEGTPDEC_SRC = '''
  libMpegTPDec/src/tpdec_adif.cpp
  libMpegTPDec/src/tpdec_adts.cpp
  libMpegTPDec/src/tpdec_asc.cpp
  libMpegTPDec/src/tpdec_drm.cpp
  libMpegTPDec/src/tpdec_latm.cpp
  libMpegTPDec/src/tpdec_lib.cpp
  libMpegTPDec/src/tp_version.h
  libMpegTPDec/src/tpdec_adif.h
  libMpegTPDec/src/tpdec_adts.h
  libMpegTPDec/src/tpdec_drm.h
  libMpegTPDec/src/tpdec_latm.h'''.split()

PCMUTILS_SRC = '''
  libPCMutils/src/limiter.cpp
  libPCMutils/src/pcm_utils.cpp
  libPCMutils/src/pcmdmx_lib.cpp
  libPCMutils/src/version.h'''.split()

SACDEC_SRC = '''
  libSACdec/src/sac_bitdec.cpp
  libSACdec/src/sac_calcM1andM2.cpp
  libSACdec/src/sac_dec.cpp
  libSACdec/src/sac_dec_conceal.cpp
  libSACdec/src/sac_dec_lib.cpp
  libSACdec/src/sac_process.cpp
  libSACdec/src/sac_qmf.cpp
  libSACdec/src/sac_reshapeBBEnv.cpp
  libSACdec/src/sac_rom.cpp
  libSACdec/src/sac_smoothing.cpp
  libSACdec/src/sac_stp.cpp
  libSACdec/src/sac_tsd.cpp
  libSACdec/src/sac_bitdec.h
  libSACdec/src/sac_calcM1andM2.h
  libSACdec/src/sac_dec.h
  libSACdec/src/sac_dec_conceal.h
  libSACdec/src/sac_dec_interface.h
  libSACdec/src/sac_dec_ssc_struct.h
  libSACdec/src/sac_process.h
  libSACdec/src/sac_qmf.h
  libSACdec/src/sac_reshapeBBEnv.h
  libSACdec/src/sac_rom.h
  libSACdec/src/sac_smoothing.h
  libSACdec/src/sac_stp.h
  libSACdec/src/sac_tsd.h'''.split()

SBRDEC_SRC = '''
  libSBRdec/src/HFgen_preFlat.cpp
  libSBRdec/src/env_calc.cpp
  libSBRdec/src/env_dec.cpp
  libSBRdec/src/env_extr.cpp
  libSBRdec/src/hbe.cpp
  libSBRdec/src/huff_dec.cpp
  libSBRdec/src/lpp_tran.cpp
  libSBRdec/src/psbitdec.cpp
  libSBRdec/src/psdec.cpp
  libSBRdec/src/psdec_drm.cpp
  libSBRdec/src/psdecrom_drm.cpp
  libSBRdec/src/pvc_dec.cpp
  libSBRdec/src/sbr_deb.cpp
  libSBRdec/src/sbr_dec.cpp
  libSBRdec/src/sbr_ram.cpp
  libSBRdec/src/sbr_rom.cpp
  libSBRdec/src/sbrdec_drc.cpp
  libSBRdec/src/sbrdec_freq_sca.cpp
  libSBRdec/src/sbrdecoder.cpp
  libSBRdec/src/env_calc.h
  libSBRdec/src/env_dec.h
  libSBRdec/src/env_extr.h
  libSBRdec/src/hbe.h
  libSBRdec/src/HFgen_preFlat.h
  libSBRdec/src/huff_dec.h
  libSBRdec/src/lpp_tran.h
  libSBRdec/src/psbitdec.h
  libSBRdec/src/psdec.h
  libSBRdec/src/psdec_drm.h
  libSBRdec/src/pvc_dec.h
  libSBRdec/src/sbr_deb.h
  libSBRdec/src/sbr_dec.h
  libSBRdec/src/sbr_ram.h
  libSBRdec/src/sbr_rom.h
  libSBRdec/src/sbrdec_drc.h
  libSBRdec/src/sbrdec_freq_sca.h
  libSBRdec/src/transcendent.h'''.split()

SYS_SRC = '''
  libSYS/src/genericStds.cpp
  libSYS/src/syslib_channelMapDescr.cpp'''.split()

sources = FDK_SRC + AACDEC_SRC + ARITHCODING_SRC + DRCDEC_SRC + MPEGTPDEC_SRC + PCMUTILS_SRC + SACDEC_SRC + SBRDEC_SRC + SYS_SRC

libAAC = static_library (
	'AAC',
	sources : sources,
	include_directories : ['libAACdec/include', 'libArithCoding/include', 'libDRCdec/include', 'libFDK/include', 'libMpegTPDec/include',
	'libPCMutils/include', 'libSACdec/include', 'libSBRdec/include', 'libSYS/include', melder_inc]
)

libAAC_dep = declare_dependency (
	include_directories : include_directories('.'), link_with: libAAC
)
